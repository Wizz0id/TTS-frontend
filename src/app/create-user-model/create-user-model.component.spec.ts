import { ComponentFixture, TestBed } from '@angular/core/testing';

import { CreateUserModelComponent } from './create-user-model.component';

describe('CreateUserModelComponent', () => {
  let component: CreateUserModelComponent;
  let fixture: ComponentFixture<CreateUserModelComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [CreateUserModelComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(CreateUserModelComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
